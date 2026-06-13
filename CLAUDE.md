# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A "thinking radar" for a folder of markdown notes. It surfaces recurring themes, BERTopic clusters, strong quotes, tensions, time-series spikes, stale-but-recurring ideas, emotional tone, and (optionally) LLM-extracted article candidates. Five milestones are complete (ingest → keywords → embeddings → clustering → LLM enrichment), and four output formats ship side-by-side: markdown report, JSON, self-contained HTML dashboard, and rich-CLI terminal summary. Tier 1 is mostly shipped: T1.1 (tag-weighted clustering + tag-aware LLM prompt), T1.1a (reconciled tag-string/tag-matrix design), T1.2 (`tta promote` action), and T1.3 (`tta diff` multi-run comparison). Remaining Tier 1 items: T1.1b (per-tag weights), T1.2a (smarter section routing for promote), T1.4 (real zero-dep keyword fallback — see `docs/FOLLOW_UP.md`).

See `README.md` for the user-facing surface (CLI flags, env vars, config file, output formats) and `docs/FOLLOW_UP.md` for the Tier 1/2/3 next-step ideas and the known limitations of the current run.

## Commands

All commands assume Windows PowerShell and the venv at `.venv\Scripts\python.exe`. `PYTHONPATH=src` is required because the package is `src/`-layout and not installed into the venv for tests in every environment.

```bash
# Activate venv (one-time per shell)
.venv\Scripts\activate

# Run the full test suite (heavy-deps tests are gated with pytest.importorskip)
PYTHONPATH=src .venv\Scripts\python.exe -m pytest -q

# Run a single test file
PYTHONPATH=src .venv\Scripts\python.exe -m pytest tests/test_m2.py -q

# Run a single test by name
PYTHONPATH=src .venv\Scripts\python.exe -m pytest tests/test_m2.py -k test_clustering_smoke -q

# Lint
ruff check src\ tests\

# Type-check (optional, not in dev extras yet)
mypy src\

# CLI — the two entry points in pyproject.toml are `text-analyzer` and `tta`
# (both map to text_theme_analyzer.cli:main).
.venv\Scripts\python.exe -m text_theme_analyzer analyze . -o markdown,html

# Generate the 19-note sample corpus (mirrors idea.txt themes)
python scripts\make_sample_notes.py
```

There is no `Makefile`, no `tox`, no `nox`. Test config lives in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Architecture

The package is `src/text_theme_analyzer/`, split into four cooperating subsystems:

```
src/text_theme_analyzer/
├── cli.py                 # Click entry point; subcommand `analyze`
├── config.py              # CLI > YAML > env > defaults precedence; tiny .env loader
├── pipeline/              # Deterministic stages; no LLM calls here
│   ├── ingest.py          # walk folder, parse frontmatter, match include/exclude globs
│   ├── csv_ingest.py      # CSV → Note (e.g. content review queues)
│   ├── preprocess.py      # strip code/HTML, chunk long bodies
│   ├── keywords.py        # YAKE / KeyBERT keyphrase extraction
│   ├── embeddings.py      # sentence-transformers + content-addressed disk cache
│   ├── clustering.py      # UMAP 2D + HDBSCAN + c-TF-IDF (BERTopic). Tag-weighted variant: pass tag_weight + tag_matrix to nudge clustering with the user's tag vocabulary (T1.1).
│   ├── timeseries.py      # weekly bucketing, spike and stale detection
│   ├── tone.py            # lightweight valence/arousal lexicon
│   ├── model.py           # dataclasses: Note, Chunk, ClusterResult, Analysis
│   └── orchestrator.py    # run(config) → Analysis; wires all stages in order
├── llm/                   # All LLM I/O lives here; the pipeline doesn't import this
│   ├── base.py            # LLMClient Protocol + LLMParseError
│   ├── ollama.py          # Ollama Pro / cloud client
│   ├── openai_compat.py   # OpenAI / Groq / Together client
│   ├── factory.py         # env-var-aware client builder
│   ├── prompts.py         # SYSTEM_PROMPT + build_user_prompt
│   ├── schemas.py         # Pydantic models for the LLM response
│   └── enrichment.py      # bundle → LLM → parse → validate → retry → post-validate
├── output/                # Four renderers, all consuming the same Analysis
│   ├── markdown_report.py
│   ├── json_report.py
│   ├── html_dashboard.py  # Jinja2 template at output/templates/dashboard.html.j2
│   ├── cli_summary.py     # rich-rendered terminal output
│   ├── history.py         # run-snapshot writer (see "Run history" below)
│   ├── diff_dashboard.py  # T1.3: `tta diff --html` output
│   └── promote.py         # T1.2: `tta promote` stub writer
└── utils/
    ├── dates.py           # tolerant frontmatter date parsing
    ├── hashing.py         # content-addressed cache keys
    └── progress.py        # tqdm wrapper, respects --quiet
```

### The single contract: `Analysis`

`pipeline/orchestrator.py::run(config) -> Analysis` is the one place where every stage is wired. Every output renderer consumes an `Analysis`. The LLM stage is bolted on at the end (in `enrichment.py`) and mutates the `Analysis` in place by attaching `EnrichmentResult`s to clusters. The four output renderers do not know whether LLM enrichment ran — they render whatever is on the `Analysis`.

### LLM enrichment contract — read this before touching `llm/`

The LLM stage is **on by default** and **degrades gracefully**: if the call fails (no key, network, parse error), `enrichment.py` logs a warning once and the pipeline still writes the no-LLM artifacts. Do not raise out of `enrichment.py` on a transient LLM failure. Quote validation is strict — invented or paraphrased quotes returned by the model are dropped during post-validation against the input pool, not silently kept. The user has been bitten before by the opposite behaviour.

### Configuration precedence

`config.py` documents it at the top of the file: **CLI flags > YAML config > env vars > hard-coded defaults**. The YAML config is auto-discovered (`text-theme-analyzer.yml` next to the input folder, then `~/.config/text-theme-analyzer/config.yml`) or passed via `--config`. The `.env` loader in `config.py` is intentionally a no-dep implementation and does **not** override pre-existing process env vars (shell exports win over the file).

### Run history

Every run writes a snapshot to `{output_dir}/run-history/{timestamp}.json` via `output/history.py`. Two CLI subcommands consume the snapshots: `tta runs` (list, oldest-first) and `tta diff OLD NEW` (multi-run comparison with IDF-weighted cluster matching, optional `--html PATH` output, optional `--match-threshold FLOAT` knob, see `output/diff_dashboard.py` and `output/templates/diff_dashboard.html.j2`). When changing the run snapshot schema, keep the JSON additive — old snapshots should still load.

## Conventions specific to this project

- **Frontmatter date resolution priority** (do not reorder, downstream code depends on the order): `date` → `created` → `published` → filename regex `YYYY-MM-DD` → file mtime → `None`. A note with no resolvable date is still clustered but excluded from time-series features.
- **Globs** in `--include` / `--exclude` are matched with a hand-rolled `fnmatch` normalizer in `ingest.py` (strips leading `**/`, matches both name and relpath). Don't swap it for `pathlib.PurePath.match` — that breaks on Windows path separators. There is a dedicated regression test at `tests/test_cli_glob_recovery.py`.
- **Schemas are a contract.** `llm/schemas.py` Pydantic models define the post-validation shape of the LLM response. Field length caps were loosened in commit `a01442d` (60/80/600 → 120/160/1000) — the real-corpus run hit those ceilings. Bump caps defensively, but think about it.
- **Outputs are additive.** Each output format is independent; removing one renderer should not affect the others. `cli.py` is the only place that fans out to the four of them.
- **The test suite is the spec.** Heavy-deps tests use `pytest.importorskip` so the suite runs even without the M2 stack (sentence-transformers, BERTopic, etc.) installed. The keyword extractor defaults to `method="keybert"` and falls back to a pure-Python zero-dep extractor when KeyBERT is unavailable, so end-to-end tests no longer need a `yake` gate (T1.4 is shipped).

## What is private and what is committed

`.gitignore` is load-bearing. These are intentionally **not** in git:

- `data/` — the author's full Obsidian vault used for real-corpus runs
- `text-theme-analyzer-sample-*/` — date-stamped private corpora (the 2026-06-07 sample is one of these)
- `text-theme-output/`, `text-theme-output-llm/`, `out/`, `reports/` — generated artifacts
- `text-theme-analyzer.yml`, `text-theme-analyzer.local.yml` — local user config tuned to a specific corpus
- `.env`, `.env.*` — secrets (only `.env.example` is committed)
- `.claude/` — local Claude Code settings (notably the `permissions.allow` allowlist)

If you add a new sample corpus, give it a `text-theme-analyzer-sample-YYYY-MM-DD` name. The `text-theme-analyzer-sample-*/` gitignore pattern is a safety net so a future folder of that shape can never be committed by accident. **Do not** loosen these patterns without checking with the user.

The CI workflows at `.github/workflows/secret-scan.yml` (gitleaks, over full history on PRs and every push to main) and `.github/workflows/test.yml` (pytest + ruff on a 3.11/3.12 × ubuntu/windows matrix, on every PR and push to main) are the second line of defence behind the `.gitignore` patterns. Gitleaks has caught nothing in practice, which is the desired outcome.

## How to add a new pipeline stage

There is no plugin registry. To add a stage, you:

1. Add a new module under `pipeline/` (e.g. `pipeline/my_stage.py`).
2. Add a dataclass to `pipeline/model.py` (or extend an existing one).
3. Wire the call in `orchestrator.py::run`, between existing stages, with a `log(...)` line that follows the `[stage] message` convention.
4. Surface the result in the four output renderers as needed. The HTML dashboard uses `dashboard.html.j2`; the markdown report is the lowest-friction place to start.
5. Add tests under `tests/`, named `test_m{N}.py` for backward compatibility with the milestone convention (the file is just a name, the suite is one flat `pytest` run).

## What is not in this repo

- No `Makefile`, `tox.ini`, `noxfile.py`, or `pyproject.toml` `[tool.poetry]` section. Build backend is `setuptools`.
- No `mypy` in the `[dev]` extras (it's listed in the README's optional section but not declared in `pyproject.toml`).
- No `pre-commit` config installed locally. `.pre-commit-config.yaml` is in the repo and ready to use, but `pre-commit install` is not run as part of any dev setup. Local development relies on the developer running `ruff` manually; the CI workflow is the enforcement.
- No `CLAUDE.md` parent / nested rules. This file is the only one.
- No docs site. `docs/` contains only `FOLLOW_UP.md`. The README is the canonical reference.
