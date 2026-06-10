# Text Theme Analyzer

A private thinking radar for your notes. Point it at a folder of markdown
files and it surfaces the themes, clusters, strong quotes, tensions, and
stale-but-recurring ideas you keep circling around.

> **Status:** all 5 milestones done. Ingest, keyphrases, embeddings, BERTopic
> clustering, time-series spikes/stale, emotional tone, LLM enrichment (Ollama
> or any OpenAI-compatible endpoint), and all four output formats
> (markdown, JSON, HTML dashboard, rich CLI) work end-to-end.

## What it surfaces

- **Recurring topics** — keyphrase document frequency
- **Clusters** — BERTopic groups with c-TF-IDF keywords and representative notes
- **Tensions** — opposing pulls between clusters (LLM-extracted)
- **Strong quotes** — quotable sentences, surfaced verbatim (LLM picks + validated)
- **Spikes** — clusters with unusually high activity in a recent week
- **Stale-but-recurring** — themes that used to fire but have gone quiet
- **Article candidates** — pull-quote-able angles extracted by the LLM
- **Emotional tone over time** — valence/arousal by month (lightweight lexicon)
- **Cluster map** — UMAP 2D projection for visual exploration

## Quick start

```bash
# 1. Install (one-time)
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# 2. Generate 19 fake notes (mirrors the themes in idea.txt)
python scripts\make_sample_notes.py

# 3. Run on them — all four output formats
python -m text_theme_analyzer analyze scripts\sample_notes -o markdown,cli,json,html --output-dir .\out --no-llm
```

You'll see `themes-report.md`, `themes.json`, `dashboard.html`, and a rich
terminal summary in `.\out\`.

## LLM enrichment

The LLM stage is **on by default** and **gracefully degrades** if the
endpoint is unreachable. To enable it:

```bash
# Ollama Pro / Ollama Cloud (default)
set TEXTHEME_OLLAMA_API_KEY=your-key-here
set TEXTHEME_OLLAMA_MODEL=minimax-m3
python -m text_theme_analyzer analyze .\notes -o markdown,html

# Any OpenAI-compatible endpoint (OpenAI, Groq, Together, etc.)
set TEXTHEME_OPENAI_COMPAT_BASE_URL=https://api.openai.com/v1
set TEXTHEME_OPENAI_COMPAT_API_KEY=sk-...
set TEXTHEME_OPENAI_COMPAT_MODEL=gpt-4o-mini
python -m text_theme_analyzer analyze .\notes --provider openai_compat -o markdown,html
```

The Ollama Cloud API lives at `https://ollama.com` and exposes both the
native `/api/chat` and the OpenAI-compatible `/v1/chat/completions`
endpoints. Default base URL: `https://ollama.com`. List the cloud catalog
at <https://ollama.com/library> to pick a model name — common picks are
`minimax-m3`, `gpt-oss:20b`, `gpt-oss:120b`, `kimi-k2.6`,
`deepseek-v3.2`. Note: don't include `:cloud` — that's only the suffix
the local `ollama run` CLI shows, not a real model tag.

If the LLM call fails (no key, network down, 4xx response), the pipeline
still produces all other outputs and logs a warning.

## CLI

```bash
python -m text_theme_analyzer analyze [OPTIONS] INPUT_PATH
```

| Flag | Description |
|---|---|
| `-o, --output` | `markdown,json,html,cli` — comma-separated, default `markdown` |
| `--output-dir` | Where written artifacts go (default `./text-theme-output`) |
| `--provider` | `ollama` (default) or `openai_compat` |
| `--model` | LLM model name (provider-specific, default `gpt-oss:20b`) |
| `--embedding-model` | sentence-transformers model (default `all-MiniLM-L6-v2`) |
| `--include` / `--exclude` | Repeatable glob patterns (default `**/*.md`, `**/*.markdown`) |
| `--since` / `--until` | ISO date filter on frontmatter/filename date |
| `--top-n-themes` | How many themes to surface (default 15) |
| `--top-n-quotes` | Strong quotes per cluster (default 5) |
| `--min-cluster-size` | HDBSCAN `min_cluster_size` override. Default uses a corpus-size heuristic. Lower = more, smaller clusters. Higher = fewer, larger clusters. |
| `--umap-n-neighbors` | UMAP `n_neighbors` for the clustering projection. Lower = more local structure, more clusters. Higher = more global, fewer clusters. |
| `--tag-weight` | Tag-weighted clustering weight (T1.1, default `0.0` = off). When `> 0`, concatenates a one-hot over the corpus's top-N tags to each chunk's embedding, scaled by this weight, before clustering. 0.3-0.5 is a reasonable starting point for personal-vault corpora. |
| `--top-n-tags` | Cap on the global tag vocabulary used for tag-weighted clustering and the LLM tag-distribution prompt (T1.1, default `20`). |
| `--spike-window-weeks` | Rolling window for spike detection (default 8) |
| `--stale-window-weeks` | Recent-quiet threshold for stale ideas (default 8) |
| `--no-llm` | Skip LLM enrichment (fast, deterministic) |
| `--no-cache` | Bypass the disk embedding cache |
| `--cache-dir` | Where embeddings are cached (default `~/.cache/text-theme-analyzer`) |
| `--config` | Path to a YAML config file (auto-discovered otherwise) |
| `-v, --verbose` / `-q, --quiet` | Logging knobs |

## Promote to project (T1.2)

When the LLM returns a `promote_to_project` verdict on a stale cluster,
it copies a `tta promote <promote_key> ...` invocation to the clipboard.
Pasting and running that command writes a pre-filled project stub to a
markdown file you control.

```bash
# Default target: ./promoted-projects.md (overridable in YAML).
# Default source: most recent run in ./text-theme-output.
tta promote 12:2026-04-15

# Override the source run or target file from the CLI.
tta promote 12:2026-04-15 --from-run ./runs/2026-06-10
tta promote 12:2026-04-15 --target-file ./projects/agency.md
```

The target file is a flat list of `## ` buckets (e.g. `## Promoted`),
each containing one or more `### ` project stubs. Re-running
`tta promote` for the same `promote_key` replaces that stub in place;
a new key appends. This is the design's idempotency guarantee.

```yaml
# text-theme-analyzer.yml — promote config
promote:
  target_file: ./promoted-projects.md
  # Optional Kanban-style buckets. The first entry is the default
  # landing bucket; pass --section to override per-invocation.
  sections:
    - To start
    - In progress
    - Archive
```

The output is plain markdown — standard `[text](path)` links, no
`[[wikilinks]]`, no plugin-specific syntax. The Obsidian Kanban plugin
will render the `## ` buckets as columns for free, but no Obsidian
install is required to read or write the file.

## Multi-run diff (T1.3)

Every `analyze` run writes a snapshot to `{output_dir}/run-history/{ISO-timestamp}.json`.
`tta diff` compares two of them and shows what changed between runs.

```bash
# Text diff to stdout.
tta diff 2026-06-01T10-00-00Z 2026-06-10T10-00-00Z

# Also write a 2-column side-by-side HTML dashboard.
tta diff 2026-06-01T10-00-00Z 2026-06-10T10-00-00Z --html ./runs/diff.html

# Stricter matching (default 0.3). Higher = fewer matches.
tta diff OLD NEW --match-threshold 0.5
```

The text diff reports clusters that were **added** (no match in old),
**removed** (no match in new), **grew** / **shrank** (matched pair
with size delta), and **stable** (matched pair, same size). It
also reports new and dropped top keyphrases, plus spike and stale
deltas.

The HTML dashboard is a self-contained static page with a summary
block, a matched-pairs table (one row per matched pair, columns:
similarity, old cluster, new cluster, size delta, verdict pill),
and separate sections for added and removed clusters. No JS, no
external CSS, safe to email.

**How matching works:** clusters are matched by IDF-weighted
cosine similarity of their top-8 c-TF-IDF keywords (the snapshot's
`cluster_fingerprints` field). BERTopic reassigns cluster IDs per
run, so matching by raw `cid` would report every cluster as
added+removed; the fingerprint catches the actual theme identity.
Below `--match-threshold` (default `0.3`), a pair is considered
"not the same theme" and the new cluster is `added` (or the old
is `removed`).

Pre-T1.3 snapshots (without the `cluster_fingerprints` field) fall
back to raw-ID matching, so old runs and old diffs keep working.

Use `tta runs` to list available snapshot timestamps.

## Config file (optional)

`text-theme-analyzer.yml` next to the input folder, or
`~/.config/text-theme-analyzer/config.yml`. CLI flags > config > env > defaults.

```yaml
provider: ollama
model: gpt-oss:20b
embedding_model: all-MiniLM-L6-v2
ollama:
  base_url: https://ollama.com
  api_key_env: TEXTHEME_OLLAMA_API_KEY
  timeout_s: 120
outputs: [markdown, html, cli]
output_dir: ./reports
top_n_themes: 20
spike_window_weeks: 8
stale_window_weeks: 8
promote:
  target_file: ./promoted-projects.md
  sections:
    - To start
    - In progress
    - Archive
exclude:
  - "**/_archive/**"
  - "**/templates/**"
```

## Env vars (all `TEXTHEME_` prefixed)

| Variable | Default | Purpose |
|---|---|---|
| `TEXTHEME_OLLAMA_API_KEY` | (none) | Ollama Pro / cloud key |
| `TEXTHEME_OLLAMA_BASE_URL` | `https://ollama.com` | Endpoint root |
| `TEXTHEME_OLLAMA_MODEL` | `gpt-oss:20b` | Default model |
| `TEXTHEME_OLLAMA_TIMEOUT` | `120` | Per-request timeout (s) |
| `TEXTHEME_OPENAI_COMPAT_BASE_URL` | (none) | e.g. `https://api.openai.com/v1` |
| `TEXTHEME_OPENAI_COMPAT_API_KEY` | (none) | API key for the openai_compat endpoint |
| `TEXTHEME_OPENAI_COMPAT_MODEL` | (none) | Defaults to the value of `--model` |
| `TEXTHEME_CACHE_DIR` | `~/.cache/text-theme-analyzer` | Embedding cache root |
| `TEXTHEME_LOG_LEVEL` | `INFO` | Standard levels |

## Frontmatter

The tool parses YAML frontmatter. Recognized fields:

```yaml
---
date: 2025-04-01                # or created / published
title: My note title
tags: [ai, agents, workflow]    # list or comma-separated string
---
```

Date resolution priority: frontmatter `date` → `created` → `published` →
filename regex `YYYY-MM-DD` → file mtime → `None` (note is still
clustered but excluded from time-series).

The `tags` list (frontmatter and/or inline `#tag`) feeds the LLM
enrichment prompt as a per-cluster tag distribution. With
`--tag-weight > 0`, the same tags also nudge clustering: two notes
that share tags get pulled toward the same cluster even when their
prose is dissimilar. `0.0` (the default) keeps the pre-T1.1 behavior.

## CSV input

CSV files (e.g. a content review queue) are supported as a first-class
input format. They are **not** matched by the default `--include` glob —
opt in with `--include "**/*.csv"`. Each non-empty row becomes a Note:

- **title** is `"[<id>] <theme>"`
- **date** comes from `date_created` (overridable)
- **tags** are `theme` + `platform`
- **body** is a labeled multi-section block: `Theme`, `Source`,
  `Platform`, `Format`, `Tone`, `Personal level`, `Private risk`,
  `Status`, `Reuse as`, `Draft`, `Hook`, `Your comment`,
  `Notes for revision`. Empty fields are dropped.

```bash
# A folder that mixes markdown notes and a content review CSV
python -m text_theme_analyzer analyze .\content-pipeline \
    --include "**/*.md" --include "**/*.csv" -o markdown
```

The default column names match the schema used by the bundled sample
content review queue. To ingest a CSV with a different header layout,
call `text_theme_analyzer.pipeline.csv_ingest.load_csv(path, column_map={...})`
directly from a tiny script — see the module docstring for the logical
field names.

## Performance

For ~200 notes (~1.5KB each) on a laptop CPU:

| Stage | Cold | Warm (cache hit) |
|---|---|---|
| Ingest + preprocess | <1s | <1s |
| KeyBERT (or YAKE) | ~5s | ~5s |
| Embedding (MiniLM, CPU) | 30-60s | <1s |
| UMAP + HDBSCAN + c-TF-IDF | ~5s | ~5s |
| Time-series + tone | <1s | <1s |
| LLM enrichment (Ollama Pro) | 15-45s | 15-45s |
| Output rendering | <1s | <1s |
| **Total** | **~60-120s** | **~20-55s** |

Bottleneck on cold runs is the embedding download (~80MB) and compute.
The cache directory holds the result so re-runs are near-instant on the
embedding stage.

## Architecture

```
src\text_theme_analyzer\
├── cli.py                       # Click entry point
├── config.py                    # env-var + YAML config loader
├── pipeline\
│   ├── ingest.py                # walk folder, parse frontmatter
│   ├── preprocess.py            # strip code, chunk long bodies
│   ├── keywords.py              # keyphrase extraction (YAKE / KeyBERT)
│   ├── embeddings.py            # sentence-transformers + disk cache
│   ├── clustering.py            # BERTopic + UMAP 2D
│   ├── timeseries.py            # weekly bucketing, spike + stale
│   ├── tone.py                  # lightweight valence/arousal
│   ├── orchestrator.py          # wires all stages
│   └── model.py                 # dataclasses
├── llm\
│   ├── base.py                  # LLMClient Protocol + exceptions
│   ├── ollama.py                # Ollama Pro client
│   ├── openai_compat.py         # OpenAI / Groq / Together client
│   ├── factory.py               # env-var-aware client builder
│   ├── enrichment.py            # bundle → LLM → validate → retry
│   ├── prompts.py               # system + user prompt templates
│   └── schemas.py               # Pydantic models for the response
├── output\
│   ├── markdown_report.py
│   ├── json_report.py
│   ├── html_dashboard.py
│   ├── cli_summary.py           # rich-rendered terminal output
│   └── templates\dashboard.html.j2
└── utils\
    ├── hashing.py               # content-addressed cache keys
    ├── dates.py                 # tolerant frontmatter date parsing
    └── progress.py              # tqdm wrapper
```

## Tests

50 tests across 5 test files; run with:

```bash
PYTHONPATH=src .venv\Scripts\python.exe -m pytest -q
```

Heavy-deps tests are gated with `pytest.importorskip` so the suite runs
even if you skip the M2 stack (sentence-transformers, BERTopic, etc.).

## Development

```bash
# Install with dev extras (pulls in ruff + pytest-cov)
pip install -e .[dev]

# Lint
ruff check src\ tests\
# Auto-format
ruff format src\ tests\

# Type-check (optional)
mypy src\
```

### Pre-commit hooks

A `.pre-commit-config.yaml` runs `ruff check --fix`, `ruff format`, and
`gitleaks` on every commit — same checks as the CI workflows, but as
local feedback. One-time setup:

```bash
pip install pre-commit
pre-commit install
```

Run on the whole tree without committing: `pre-commit run --all-files`.

## What you get

Running on the seeded sample notes produces:

- **4 clusters** with the right representatives:
  - Cluster 0: agent / model / workflow / loop → AI agents ✓
  - Cluster 1: freedom / work / hard / long → career / freedom-vs-structure ✓
  - Cluster 2: attention / thing / game / weight → Signal Lost / quiet games ✓
  - Cluster 3: grift / always / people / audience → scams (and **stale**) ✓
- **2 stale-but-recurring** ideas: cluster 3 (scams, last seen 2024-11) and cluster 2 (games, last seen 2026-01)
- **Tones over 14 months** scored via the lexicon
- A 4-quote / 4-tension LLM enrichment (if the LLM is reachable)
- A 18KB self-contained HTML dashboard

## License

MIT
