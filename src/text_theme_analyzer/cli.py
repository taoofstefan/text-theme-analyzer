"""Click-based CLI entry point.

M1 scope: single `analyze` subcommand, no LLM, markdown output.
M2+ adds: --no-llm flag, --output json,html,cli; LLM enrichment.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import click

from text_theme_analyzer import __version__
from text_theme_analyzer.config import (
    Config,
    OutputFormat,
    Provider,
    apply_env_overrides,
    apply_yaml_overrides,
    find_config_file,
    load_dotenv,
    load_yaml_config,
)
from text_theme_analyzer.output.cli_summary import render_cli
from text_theme_analyzer.output.html_dashboard import write_html
from text_theme_analyzer.output.json_report import write_json
from text_theme_analyzer.output.markdown_report import write_markdown
from text_theme_analyzer.pipeline.orchestrator import run as run_pipeline


def _parse_date(ctx, param, value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise click.BadParameter(f"Expected ISO date YYYY-MM-DD, got {value!r}") from None


def _build_config(
    *,
    cli_passed: set[str],
    input_path: Path,
    outputs: tuple[str, ...],
    output_dir: Path,
    provider: str,
    model: str,
    embedding_model: str,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    since: date | None,
    until: date | None,
    top_n_themes: int,
    top_n_quotes: int,
    no_merge_contained_phrases: bool,
    min_cluster_size: int | None,
    umap_n_neighbors: int | None,
    tag_weight: float,
    top_n_tags: int,
    tag_weights_json: str | None,
    no_llm: bool,
    dry_run: bool,
    cache_dir: Path,
    no_cache: bool,
    config_path: Path | None,
    verbose: bool,
    quiet: bool,
) -> Config:
    # Build a Config from defaults, then layer YAML < env < CLI flag overrides.
    # Resolution order (highest priority first, matches docstring at top of config.py):
    #   1. CLI flags the user actually passed on the command line
    #   2. Env vars
    #   3. YAML config file (--config, or auto-discovered)
    #   4. Hard-coded defaults
    # We detect "passed on the command line" via `cli_passed: set[str]` (populated
    # by the Click callback using `ctx.get_parameter_source`).
    config = Config()

    # Lowest priority first: YAML.
    load_dotenv()
    yaml_path = config_path or find_config_file()
    if yaml_path:
        apply_yaml_overrides(config, load_yaml_config(yaml_path))

    # Then env (a step up).
    apply_env_overrides(config)

    # Then explicit CLI flags. `cli_passed` names the flags the user actually
    # supplied on the command line — defaults like `output_dir=Path("./text-theme-output")`
    # would otherwise silently override the YAML's value, which is the bug this
    # function exists to prevent.
    if "input_path" in cli_passed:
        config.input_path = input_path
    if "output" in cli_passed:
        config.outputs = [OutputFormat(o) for o in outputs]
    if "output_dir" in cli_passed:
        config.output_dir = output_dir
    if "provider" in cli_passed:
        config.provider = Provider(provider)
    if "model" in cli_passed and model:
        config.model = model
    if "embedding_model" in cli_passed and embedding_model:
        config.embedding_model = embedding_model
    if "include" in cli_passed and include:
        config.include = list(include)
    if "exclude" in cli_passed and exclude:
        config.exclude = list(exclude)
    if "since" in cli_passed:
        config.since = since
    if "until" in cli_passed:
        config.until = until
    if "top_n_themes" in cli_passed:
        config.top_n_themes = top_n_themes
    if "top_n_quotes" in cli_passed:
        config.top_n_quotes = top_n_quotes
    if "no_merge_contained_phrases" in cli_passed:
        config.merge_contained_phrases = not no_merge_contained_phrases
    if "min_cluster_size" in cli_passed:
        config.min_cluster_size = min_cluster_size
    if "umap_n_neighbors" in cli_passed:
        config.umap_n_neighbors = umap_n_neighbors
    if "tag_weight" in cli_passed:
        config.tag_weight = tag_weight
    if "top_n_tags" in cli_passed:
        config.top_n_tags = top_n_tags
    if "tag_weights" in cli_passed and tag_weights_json is not None:
        import json
        config.tag_weights = {str(k): float(v) for k, v in json.loads(tag_weights_json).items()}
    if "no_llm" in cli_passed:
        config.no_llm = no_llm
    if "dry_run" in cli_passed:
        config.dry_run = dry_run
    if "cache_dir" in cli_passed and cache_dir is not None:
        config.cache_dir = cache_dir
    if "no_cache" in cli_passed:
        config.no_cache = no_cache
    if "verbose" in cli_passed:
        config.verbose = verbose
    if "quiet" in cli_passed:
        config.quiet = quiet

    return config


def _split_csv(ctx, param, value: tuple[str, ...]) -> tuple[str, ...]:
    """Allow `-o markdown,cli` OR `-o markdown -o cli`. Validates against OutputFormat."""
    valid = {f.value for f in OutputFormat}
    out: list[str] = []
    for v in value:
        for piece in v.split(","):
            piece = piece.strip()
            if not piece:
                continue
            if piece not in valid:
                raise click.BadParameter(
                    f"{piece!r} is not one of {sorted(valid)}"
                )
            out.append(piece)
    return tuple(out)


def _record_passed(ctx: click.Context, param: click.Parameter, value):
    """Click callback that records which options were passed on the command line.

    Used so the config layer can distinguish "user typed --foo" from "default
    value populated by Click" — only the former should win over YAML/env.
    """
    if (
        ctx is not None
        and ctx.obj is not None
        and ctx.get_parameter_source(param.name) is click.core.ParameterSource.COMMANDLINE
    ):
        ctx.obj.setdefault("cli_passed", set()).add(param.name)
    return value


_GLOB_META = set("*?[")


def _looks_like_path(token: str) -> bool:
    """True for a token that's probably a filesystem path, not a flag value
    and not a glob pattern. Used by the argv pre-processor to recover from
    the shell-expanded-glob failure mode.
    """
    if not token or token.startswith("-"):
        return False
    if any(c in _GLOB_META for c in token):
        return False
    # Has a path separator OR a known extension somewhere in the tail.
    if "/" in token or "\\" in token:
        return True
    return bool(Path(token).suffix)


def _absorb_expanded_glob_args(argv: list[str]) -> list[str]:
    """Repair argv when a shell (cmd.exe, globstar-off bash) has eagerly
    expanded a glob like `--include "**/*.md"` into a list of file paths.

    Two failure modes are handled:

    1. The shell expanded only the **value**, leaving `--include` intact:
       `--include "**/*.md" file1.md file2.md --include "**/*.csv"`
       Each path-like token after a multi-value flag is rewritten as
       its own `--include <path>` (or `--exclude <path>`) so Click's
       `multiple=True` semantics consume it as a separate value.

    2. The shell expanded the **whole** `--include "**/*.md"` sequence,
       including the flag itself, so the file paths arrive naked
       between `INPUT_PATH` and the next flag. We look for the first
       cluster of path-like tokens after `INPUT_PATH` and wrap each
       in `--include <path>`.

    Idempotent: if the user already single-quoted their glob, no
    path-like tokens appear, and argv is returned unchanged.
    """
    multi_value_flags = {"--include", "--exclude"}

    # Phase 1: fix mid-argv (flag intact, value expanded).
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in multi_value_flags and i + 1 < len(argv):
            out.append(tok)
            out.append(argv[i + 1])  # first value (the glob or explicit value)
            j = i + 2
            while j < len(argv) and _looks_like_path(argv[j]):
                out.append(tok)  # repeat the flag
                out.append(argv[j])
                j += 1
            i = j
        else:
            out.append(tok)
            i += 1

    # Phase 2: fix leading-argv (the whole `--include` was eaten).
    # Layout: [subcommand, INPUT_PATH, file1, file2, ..., --flag, ...]
    if len(out) >= 3 and out[0] in {"analyze", "diff", "runs"}:
        sub = out[0]
        # For `analyze`, INPUT_PATH is the second token.
        # For `diff` and `runs`, INPUT_PATH is positional and not relevant here.
        if sub == "analyze":
            # Walk forward from index 2; collect leading path-like tokens.
            j = 2
            while j < len(out) and _looks_like_path(out[j]):
                j += 1
            if j > 2:
                leading_paths = out[2:j]
                rest = out[j:]
                wrapped = []
                for p in leading_paths:
                    wrapped.append("--include")
                    wrapped.append(p)
                out = out[:2] + wrapped + rest

    return out


def _warn_if_glob_expanded(argv: list[str], input_path: Path) -> None:
    """If the user passed a `--include` glob that got expanded by the shell,
    log a one-line warning. Best-effort: only fires when we see
    `**/...` style patterns in the original (un-repaired) argv.
    """
    for tok in argv:
        if "**" in tok and any(c in _GLOB_META for c in tok):
            click.echo(
                f"[warn] shell expanded glob {tok!r}; consider single-quoting it "
                f"in {input_path}/ to avoid this. "
                f"The tool recovered by absorbing the expanded paths.",
                err=True,
            )
            return


@click.group()
@click.version_option(__version__, prog_name="text-analyzer")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Text Theme Analyzer — a private thinking radar for your notes."""
    # Windows defaults to cp1252 for stdout; force UTF-8 so reports work.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    # `ctx.obj` is where `_record_passed` stashes the set of flags the user
    # actually supplied on the command line. Initialize it here so callbacks
    # that fire before any subcommand body can rely on it being a dict.
    if ctx.obj is None:
        ctx.obj = {}


# Repair shell-expanded globs in the args list *before* Click parses.
# Click reads args from sys.argv when called via `python -m ...` (no
# explicit `args=`) and from the `args=` kwarg when called via
# CliRunner. We override `main()` on the group to handle both paths.
_original_main = main.main


def _patched_main(args=None, *call_args, **call_kwargs):
    if args is None:
        # Real CLI invocation: pull from sys.argv (already handled by
        # __main__.py for the `python -m ...` path, but be defensive in
        # case the user imports `main` and calls it directly).
        if getattr(main, "_glob_repair_warned", False) is False:
            main._glob_repair_warned = True  # type: ignore[attr-defined]
            original = sys.argv[1:]
            sys.argv[1:] = _absorb_expanded_glob_args(sys.argv[1:])
            if len(sys.argv) > 1:
                rest = sys.argv[1:]
                if rest and rest[0] == "analyze" and len(rest) >= 2:
                    try:
                        _warn_if_glob_expanded(original, Path(rest[1]))
                    except (OSError, ValueError):
                        pass
        return _original_main(*call_args, args=None, **call_kwargs)
    # Explicit args (e.g. CliRunner): repair in place.
    if getattr(main, "_glob_repair_warned", False) is False:
        main._glob_repair_warned = True  # type: ignore[attr-defined]
        repaired = _absorb_expanded_glob_args(list(args))
        if len(repaired) > 2 and repaired[0] == "analyze":
            try:
                _warn_if_glob_expanded(list(args), Path(repaired[1]))
            except (OSError, ValueError):
                pass
        args = repaired
    return _original_main(*call_args, args=args, **call_kwargs)


main.main = _patched_main  # type: ignore[method-assign]


@main.command()
@click.argument("input_path", type=click.Path(exists=True, file_okay=False, path_type=Path), callback=_record_passed)
@click.option(
    "-o", "--output",
    multiple=True,
    default=("markdown",),
    callback=lambda c, p, v: _split_csv(c, p, _record_passed(c, p, v)),
    help="Output format(s). Comma-separated or repeat the flag.",
)
@click.option("--output-dir", type=Path, default=Path("./text-theme-output"), show_default=True, callback=_record_passed)
@click.option("--provider", type=click.Choice([p.value for p in Provider]), default=Provider.OLLAMA.value, callback=_record_passed)
@click.option("--model", default=None, help="LLM model name (provider-specific).", callback=_record_passed)
@click.option("--embedding-model", default=None, help="sentence-transformers model.", callback=_record_passed)
@click.option("--include", multiple=True, help="Glob to include (repeatable).", callback=_record_passed)
@click.option("--exclude", multiple=True, help="Glob to exclude (repeatable).", callback=_record_passed)
@click.option("--since", default=None, callback=_parse_date, help="Earliest note date (ISO).")
@click.option("--until", default=None, callback=_parse_date, help="Latest note date (ISO).")
@click.option("--top-n-themes", type=int, default=15, show_default=True, callback=_record_passed)
@click.option("--top-n-quotes", type=int, default=5, show_default=True, callback=_record_passed)
@click.option("--no-merge-contained-phrases", "no_merge_contained_phrases", is_flag=True, callback=_record_passed,
              help="Disable the dedup of phrases that are strict substrings of a longer phrase "
                   "(e.g. drop 'discord direct' if 'discord direct conversation' also appears).")
@click.option("--min-cluster-size", type=int, default=None, callback=_record_passed,
              help="HDBSCAN min_cluster_size override. Default uses a corpus-size heuristic. "
                   "Lower = more, smaller clusters. Higher = fewer, larger clusters.")
@click.option("--umap-n-neighbors", type=int, default=None, callback=_record_passed,
              help="UMAP n_neighbors override for the clustering projection. "
                   "Lower = more local structure, more clusters. Higher = more global, fewer clusters.")
@click.option("--tag-weight", type=float, default=0.0, show_default=True, callback=_record_passed,
              help="Tag-weighted clustering weight (T1.1). 0.0 disables it (default). "
                   "Values >0 concatenate a one-hot over the corpus's top-N tags to each chunk's "
                   "embedding before clustering, scaled by this weight. 0.3-0.5 is a reasonable "
                   "starting point for personal-vault corpora.")
@click.option("--top-n-tags", type=int, default=20, show_default=True, callback=_record_passed,
              help="Cap on the global tag vocabulary used for tag-weighted clustering and the LLM "
                   "tag-distribution prompt (T1.1). The top-N most-frequent tags in the corpus are "
                   "retained; the rest are dropped.")
@click.option("--tag-weights", "tag_weights_json", default=None, callback=_record_passed,
              help="JSON map of tag string to multiplier for per-tag weighting (T1.1b). "
                   "Example: '{\"consulting\":2.0,\"life\":0.5}'. Applied column-wise before --tag-weight.")
@click.option("--no-llm", is_flag=True, help="Skip LLM enrichment (M1 default behavior).", callback=_record_passed)
@click.option("--dry-run", is_flag=True, help="Run ingest + clustering, then print a preview (LLM bundle size estimate) and exit. No LLM call, no output files written.", callback=_record_passed)
@click.option("--cache-dir", type=Path, default=None, callback=_record_passed)
@click.option("--no-cache", is_flag=True, callback=_record_passed)
@click.option("--config", "config_path", type=Path, default=None)
@click.option("-v", "--verbose", is_flag=True, callback=_record_passed)
@click.option("-q", "--quiet", is_flag=True, callback=_record_passed)
@click.option("--no-history", is_flag=True, help="Skip writing a run snapshot to {output_dir}/run-history/.")
def analyze(
    input_path: Path,
    output: tuple[str, ...],
    output_dir: Path,
    provider: str,
    model: str | None,
    embedding_model: str | None,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    since: date | None,
    until: date | None,
    top_n_themes: int,
    top_n_quotes: int,
    no_merge_contained_phrases: bool,
    min_cluster_size: int | None,
    umap_n_neighbors: int | None,
    tag_weight: float,
    top_n_tags: int,
    tag_weights_json: str | None,
    no_llm: bool,
    dry_run: bool,
    cache_dir: Path | None,
    no_cache: bool,
    config_path: Path | None,
    verbose: bool,
    quiet: bool,
    no_history: bool,
) -> None:
    """Analyze a folder of notes and emit theme reports."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    # Silence the very chatty httpx logs from huggingface_hub during model download.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    cli_passed: set[str] = click.get_current_context().obj.setdefault("cli_passed", set())
    config = _build_config(
        cli_passed=cli_passed,
        input_path=input_path,
        outputs=output,
        output_dir=output_dir,
        provider=provider,
        model=model or "gpt-oss:20b",
        embedding_model=embedding_model or "all-MiniLM-L6-v2",
        include=include,
        exclude=exclude,
        since=since,
        until=until,
        top_n_themes=top_n_themes,
        top_n_quotes=top_n_quotes,
        no_merge_contained_phrases=no_merge_contained_phrases,
        min_cluster_size=min_cluster_size,
        umap_n_neighbors=umap_n_neighbors,
        tag_weight=tag_weight,
        top_n_tags=top_n_tags,
        tag_weights_json=tag_weights_json,
        no_llm=no_llm,
        dry_run=dry_run,
        cache_dir=cache_dir or Path.home() / ".cache" / "text-theme-analyzer",
        no_cache=no_cache,
        config_path=config_path,
        verbose=verbose,
        quiet=quiet,
    )

    analysis = run_pipeline(config)

    # Dry-run: print a tidy preview and exit before writing any output files.
    if config.dry_run:
        preview = analysis.metadata.get("dry_run", {})
        click.echo("")
        click.echo("Dry run preview")
        click.echo("================")
        click.echo(f"  Notes:               {len(analysis.notes)}")
        click.echo(f"  Chunks:              {len(analysis.chunks)}")
        if analysis.clusters is not None:
            click.echo(f"  Clusters:            {len(analysis.clusters.cluster_keywords)}")
            click.echo(f"  Outliers:            {analysis.clusters.outlier_count}")
        if "date_range" in analysis.metadata:
            lo, hi = analysis.metadata["date_range"]
            click.echo(f"  Date range:          {lo} -> {hi}")
        if preview:
            click.echo(f"  LLM bundle (est):    ~{preview.get('est_tokens', 0):,} tokens")
            click.echo(f"    clusters in bundle: {preview.get('cluster_count', 0)}")
            click.echo(f"    excerpt chars:     {preview.get('excerpt_chars', 0):,}")
        click.echo("")
        click.echo("(no output files written, no LLM call made)")
        return

    written: list[Path] = []
    for fmt in config.outputs:
        if fmt == OutputFormat.MARKDOWN:
            path = write_markdown(analysis, config.output_dir, top_n_themes=config.top_n_themes)
            written.append(path)
        elif fmt == OutputFormat.CLI:
            render_cli(analysis, top_n_themes=config.top_n_themes)
        elif fmt == OutputFormat.JSON:
            path = write_json(analysis, config.output_dir)
            written.append(path)
        elif fmt == OutputFormat.HTML:
            path = write_html(analysis, config.output_dir)
            written.append(path)

    for path in written:
        click.echo(f"wrote {path}")

    # Persist a compact run snapshot for later `tta diff` calls.
    # Skipped on dry-run (we returned earlier) and when --no-history was set.
    if not no_history:
        from text_theme_analyzer.output.history import (
            snapshot_from_analysis,
            write_snapshot,
        )
        snap = snapshot_from_analysis(analysis)
        snap_path = write_snapshot(snap, config.output_dir)
        click.echo(f"snapshot {snap_path.name}")


@main.command("diff")
@click.argument("old", metavar="OLD_TIMESTAMP")
@click.argument("new", metavar="NEW_TIMESTAMP")
@click.option(
    "--output-dir", type=Path, default=Path("./text-theme-output"), show_default=True,
    help="Where to find the run-history/ directory.",
)
@click.option(
    "--html", "html_out", type=Path, default=None,
    help="Also write a 2-column HTML dashboard to this path (in addition to the text diff on stdout).",
)
@click.option(
    "--match-threshold", type=float, default=0.3, show_default=True,
    help="Cosine-similarity threshold for matching clusters across runs. "
         "Pairs below this are treated as 'not the same theme' (added / removed). "
         "Lower = more aggressive matching; higher = stricter. Range 0.0-1.0.",
)
def diff_runs(
    old: str, new: str, output_dir: Path,
    html_out: Path | None, match_threshold: float,
) -> None:
    """Diff two runs of the analyzer.

    OLD_TIMESTAMP and NEW_TIMESTAMP are filename stems under
    ``{output_dir}/run-history/`` (the part before ``.json``, e.g.
    ``2026-06-07T19-42-11Z``). Use ``tta runs`` to list available snapshots.

    The text diff is always printed to stdout. Pass ``--html PATH`` to
    also write a self-contained side-by-side HTML dashboard.

    Clusters are matched between runs by IDF-weighted cosine similarity
    of their top-8 c-TF-IDF keywords (the fingerprint stored in each
    snapshot). See ``output/history.py::_match_clusters``.
    """
    from text_theme_analyzer.output.history import (
        HISTORY_DIRNAME,
        diff_snapshots,
        list_snapshots,
        load_snapshot,
        render_diff,
    )
    history_dir = output_dir / HISTORY_DIRNAME
    old_path = history_dir / f"{old}.json"
    new_path = history_dir / f"{new}.json"
    if not old_path.is_file():
        available = [p.stem for p in list_snapshots(output_dir)]
        available_s = ", ".join(available[-5:]) if available else "(none)"
        raise click.BadParameter(
            f"snapshot {old!r} not found in {history_dir}. Available: {available_s}"
        )
    if not new_path.is_file():
        available = [p.stem for p in list_snapshots(output_dir)]
        available_s = ", ".join(available[-5:]) if available else "(none)"
        raise click.BadParameter(
            f"snapshot {new!r} not found in {history_dir}. Available: {available_s}"
        )
    old_snap = load_snapshot(old_path)
    new_snap = load_snapshot(new_path)
    diff = diff_snapshots(old_snap, new_snap, threshold=match_threshold)
    click.echo(render_diff(diff, old=old_snap, new=new_snap))

    if html_out is not None:
        from text_theme_analyzer.output.diff_dashboard import render_diff_html
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(
            render_diff_html(diff, old=old_snap, new=new_snap),
            encoding="utf-8",
        )
        click.echo(f"wrote {html_out}", err=True)


@main.command("runs")
@click.option(
    "--output-dir", type=Path, default=Path("./text-theme-output"), show_default=True,
)
def list_runs(output_dir: Path) -> None:
    """List all stored run snapshots (oldest first)."""
    from text_theme_analyzer.output.history import list_snapshots
    paths = list_snapshots(output_dir)
    if not paths:
        click.echo("(no snapshots found)")
        return
    for p in paths:
        click.echo(p.stem)


@main.command("promote")
@click.argument("promote_key", metavar="PROMOTE_KEY")
@click.option(
    "--from-run", "from_run", type=Path, default=None,
    help="Path to a run output dir (containing themes.json). Default: most recent in --output-dir.",
)
@click.option(
    "--output-dir", type=Path, default=Path("./text-theme-output"), show_default=True,
    help="Where to look for the most recent run when --from-run is not given.",
)
@click.option(
    "--target-file", "target_file", type=Path, default=None,
    help="Override the target markdown file (default: promote.target_file in config).",
)
@click.option(
    "--section", "section", default=None,
    help="Override which `## ` heading the stub is appended under (default: the first entry in promote.sections, or 'Promoted').",
)
@click.option("--config", "config_path", type=Path, default=None)
def promote_cmd(
    promote_key: str,
    from_run: Path | None,
    output_dir: Path,
    target_file: Path | None,
    section: str | None,
    config_path: Path | None,
) -> None:
    """Promote a stale-recurring verdict to a project plan file.

    PROMOTE_KEY is the value shown in the dashboard's "Copy command"
    button (format: ``<cluster_id>:<last_seen>``). The verdict is
    looked up in ``{from_run}/themes.json``; the project stub is
    appended (or replaced) in the configured target file.
    """
    from text_theme_analyzer.output.promote import (
        ClusterContext,
        apply_promotion,
        render_promote_stub,
    )

    # Load config (for promote.target_file + promote.sections defaults).
    # Mirrors the layering in _build_config(): defaults < YAML < env.
    # The promote subcommand has no `--quiet` / `--verbose` / `--no-llm` /
    # etc. options of its own, so we just need the config object, not the
    # full _build_config pipeline.
    load_dotenv()
    yaml_path = config_path or find_config_file()
    config = Config()
    if yaml_path:
        apply_yaml_overrides(config, load_yaml_config(yaml_path))
    apply_env_overrides(config)

    # Resolve --from-run: explicit > most recent in --output-dir.
    run_dir = from_run
    if run_dir is None:
        run_dir = _latest_run_dir(output_dir)
        if run_dir is None:
            raise click.UsageError(
                f"No run found in {output_dir}. Re-run the analyzer with an LLM, or pass --from-run."
            )
    themes_json = run_dir / "themes.json"
    if not themes_json.is_file():
        raise click.UsageError(
            f"themes.json not found at {themes_json}. Pass --from-run pointing at a run output dir."
        )

    data = json.loads(themes_json.read_text(encoding="utf-8"))
    promote_keys = data.get("promote_keys") or {}
    if promote_key not in promote_keys:
        available = ", ".join(list(promote_keys.keys())[:5]) or "(none)"
        raise click.UsageError(
            f"promote_key {promote_key!r} not found in {themes_json}. Available: {available}"
        )

    record = promote_keys[promote_key]
    if record.get("verdict") != "promote_to_project":
        raise click.UsageError(
            f"Verdict for {promote_key!r} is {record.get('verdict')!r}, not 'promote_to_project'. "
            "Only promote_to_project verdicts can be promoted."
        )

    # Resolve representative-note references into (date, title, path) tuples
    # for the "Supporting notes" section of the stub.
    files_by_id = {f["id"]: f for f in (data.get("files") or [])}
    rep_notes: list[tuple[str, str | None, str, str]] = []
    for nid in record.get("representative_note_ids") or []:
        f = files_by_id.get(nid)
        if f is None:
            continue
        rep_notes.append((nid, f.get("date"), f.get("title", nid), f.get("path", nid)))

    context = ClusterContext(
        cluster_id=int(record["cluster_id"]),
        theme=record["theme"],
        reasoning=record["reasoning"],
        last_seen=record.get("last_seen"),
        first_seen=record.get("first_seen"),
        frequency=record.get("frequency"),
        severity=record.get("severity"),
        keywords=list(record.get("keywords") or []),
        representative_notes=rep_notes,
    )

    stub = render_promote_stub(promote_key, context)

    # Resolve target file + section. CLI flags override LLM verdict;
    # LLM verdict overrides config default.
    target = target_file or config.promote.target_file
    target_section = section
    if target_section is None:
        llm_section = record.get("target_section")
        if llm_section is not None:
            target_section = llm_section
    if target_section is None and config.promote.sections:
        target_section = config.promote.sections[0]

    apply_promotion(
        target_path=target,
        stub=stub,
        promote_key=promote_key,
        target_section=target_section,
        configured_sections=[str(s) for s in config.promote.sections],
    )

    click.echo(f"[promote] wrote {target} ({promote_key})", err=True)


def _latest_run_dir(output_dir: Path) -> Path | None:
    """Find the most recent run directory under ``output_dir``.

    A run dir is anything directly under ``output_dir`` that contains
    a ``themes.json``. We sort by mtime descending. The check for
    ``themes.json`` (not just any dir) avoids picking up unrelated
    subdirs (``run-history/`` is excluded by this — it has snapshots,
    not themes.json).
    """
    if not output_dir.is_dir():
        return None
    candidates = [p for p in output_dir.iterdir() if p.is_dir() and (p / "themes.json").is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    sys.exit(main())
