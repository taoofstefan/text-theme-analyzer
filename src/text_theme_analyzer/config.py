"""Configuration loaded from CLI flags, YAML file, and env vars.

Resolution order (highest priority first):
1. CLI flags
2. YAML config file (--config, or auto-discovered)
3. Environment variables (TEXTHEME_*)
4. Hard-coded defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


def load_dotenv(path: Path | None = None, *, override: bool = False) -> list[str]:
    """Tiny .env loader (no extra dep). Returns the names of vars that were set.

    Looks for `.env` in `path` (or CWD if not given) and walks upward. Only
    exports lines of the form `KEY=value` or `KEY="value"`. Comments (`#`) and
    blank lines are ignored. By default existing process env vars are NOT
    overridden — explicit shell exports win over the file.
    """
    if path is None:
        # Search CWD then walk upward to filesystem root.
        candidates: list[Path] = []
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            candidates.append(parent / ".env")
        env_path = next((p for p in candidates if p.is_file()), None)
    else:
        env_path = path if path.is_file() else None
    if env_path is None:
        return []

    set_keys: list[str] = []
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip surrounding quotes (single or double).
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        set_keys.append(key)
    return set_keys


class OutputFormat(StrEnum):
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    CLI = "cli"


class Provider(StrEnum):
    OLLAMA = "ollama"
    OPENAI_COMPAT = "openai_compat"


DEFAULT_OLLAMA_BASE_URL = "https://ollama.com"
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "text-theme-analyzer"
DEFAULT_OUTPUT_DIR = Path("./text-theme-output")


@dataclass
class OllamaConfig:
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    api_key_env: str = "TEXTHEME_OLLAMA_API_KEY"
    timeout_s: float = 120.0


@dataclass
class OpenAICompatConfig:
    base_url_env: str = "TEXTHEME_OPENAI_COMPAT_BASE_URL"
    api_key_env: str = "TEXTHEME_OPENAI_COMPAT_API_KEY"
    model_env: str = "TEXTHEME_OPENAI_COMPAT_MODEL"


@dataclass
class PromoteConfig:
    """T1.2 — where the `tta promote` subcommand writes project stubs.

    `target_file` is the markdown file that stubs are appended to
    (default: `promoted-projects.md` next to the input folder). When
    `sections` is non-empty, new stubs are appended under the first
    matching `## ` heading (creating it if it doesn't exist). The
    "To start" / "In progress" / "Archive" convention is up to the
    user; the tool does not enforce it.
    """
    target_file: Path = field(default_factory=lambda: Path("promoted-projects.md"))
    sections: list[str] = field(default_factory=list)


@dataclass
class CorpusConfig:
    """T2.2 — one entry in the `corpora:` map for `tta run-all`.

    Each corpus overrides the global defaults it specifies. Unspecified
    fields inherit from the global `Config`. The `output_dir` default is
    `{global_output_dir}/{name}`.
    """

    input_path: Path | None = None
    output_dir: Path | None = None
    include: list[str] | None = None
    exclude: list[str] | None = None
    since: date | None = None
    until: date | None = None
    require_dates: bool | None = None
    tag_weight: float | None = None
    top_n_tags: int | None = None
    tag_weights: dict[str, float] | None = None
    min_cluster_size: int | None = None
    umap_n_neighbors: int | None = None
    promote: PromoteConfig | None = None


@dataclass
class Config:
    # I/O
    input_path: Path = field(default_factory=lambda: Path("."))
    outputs: list[OutputFormat] = field(default_factory=lambda: [OutputFormat.CLI])
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)

    # Provider
    provider: Provider = Provider.OLLAMA
    model: str = DEFAULT_OLLAMA_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openai_compat: OpenAICompatConfig = field(default_factory=OpenAICompatConfig)

    # Filtering
    include: list[str] = field(default_factory=lambda: ["**/*.md", "**/*.markdown"])
    exclude: list[str] = field(default_factory=list)
    since: date | None = None
    until: date | None = None

    # Reports
    top_n_themes: int = 15
    top_n_quotes: int = 5
    spike_window_weeks: int = 8
    stale_window_weeks: int = 8
    merge_contained_phrases: bool = True

    # Clustering (BERTopic / HDBSCAN). None = use the corpus-size heuristic
    # baked into cluster_chunks(). Override when the default is too greedy
    # (merges everything) or too eager (splits everything).
    min_cluster_size: int | None = None
    umap_n_neighbors: int | None = None

    # Tag-weighted clustering (T1.1). `tag_weight=0.0` disables it
    # (default behavior is unchanged). `top_n_tags` caps the global tag
    # vocabulary that gets one-hot-encoded and concatenated to each
    # chunk's embedding. See `pipeline/clustering.py::build_tag_matrix`.
    tag_weight: float = 0.0
    top_n_tags: int = 20
    tag_field: str = "both"  # reserved: "frontmatter" | "inline" | "both"; only "both" wired in T1.1

    # T1.1b — Optional per-tag weights. Maps a tag string to a multiplier
    # applied to that tag's column in the tag matrix before the global
    # `tag_weight` is applied. Tags not present in the map default to 1.0.
    tag_weights: dict[str, float] = field(default_factory=dict)

    # T2.1 — Require a resolvable date for every note. When True, the
    # pipeline raises before clustering if any included note has no
    # `date:` / `created:` / `published:` frontmatter and no YYYY-MM-DD
    # prefix in its filename. Off by default so existing corpora still run.
    require_dates: bool = False

    # Behavior
    no_llm: bool = False
    dry_run: bool = False
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    no_cache: bool = False

    # T1.2 — promote-to-project. See `output/promote.py`.
    promote: PromoteConfig = field(default_factory=PromoteConfig)

    # UX
    verbose: bool = False
    quiet: bool = False

    # T2.2 — per-corpus runs for `tta run-all`. Dict keys are corpus
    # names (used as URL/path slugs); values override global defaults.
    corpora: dict[str, CorpusConfig] = field(default_factory=dict)


CONFIG_LOCATIONS: list[Path] = [
    Path("./text-theme-analyzer.yml"),
    Path("./text-theme-analyzer.yaml"),
    Path.home() / ".config" / "text-theme-analyzer" / "config.yml",
]


def find_config_file() -> Path | None:
    for candidate in CONFIG_LOCATIONS:
        if candidate.exists():
            return candidate
    return None


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_yaml_overrides(config: Config, data: dict[str, Any]) -> None:
    """Mutate `config` in place from a YAML dict. Only sets keys that are present."""
    if "provider" in data:
        config.provider = Provider(data["provider"])
    if "model" in data:
        config.model = data["model"]
    if "embedding_model" in data:
        config.embedding_model = data["embedding_model"]
    if "outputs" in data:
        config.outputs = [OutputFormat(o) for o in data["outputs"]]
    if "output_dir" in data:
        config.output_dir = Path(data["output_dir"])
    if "input_path" in data:
        config.input_path = Path(data["input_path"])
    if "include" in data:
        config.include = list(data["include"])
    if "exclude" in data:
        config.exclude = list(data["exclude"])
    if "top_n_themes" in data:
        config.top_n_themes = int(data["top_n_themes"])
    if "top_n_quotes" in data:
        config.top_n_quotes = int(data["top_n_quotes"])
    if "spike_window_weeks" in data:
        config.spike_window_weeks = int(data["spike_window_weeks"])
    if "stale_window_weeks" in data:
        config.stale_window_weeks = int(data["stale_window_weeks"])
    if "merge_contained_phrases" in data:
        config.merge_contained_phrases = bool(data["merge_contained_phrases"])
    if "min_cluster_size" in data and data["min_cluster_size"] is not None:
        config.min_cluster_size = int(data["min_cluster_size"])
    if "umap_n_neighbors" in data and data["umap_n_neighbors"] is not None:
        config.umap_n_neighbors = int(data["umap_n_neighbors"])
    if "tag_weight" in data and data["tag_weight"] is not None:
        config.tag_weight = float(data["tag_weight"])
    if "top_n_tags" in data and data["top_n_tags"] is not None:
        config.top_n_tags = int(data["top_n_tags"])
    if "tag_weights" in data and isinstance(data["tag_weights"], dict):
        config.tag_weights = {str(k): float(v) for k, v in data["tag_weights"].items()}
    if "require_dates" in data:
        config.require_dates = bool(data["require_dates"])
    if "dry_run" in data:
        config.dry_run = bool(data["dry_run"])
    if "no_llm" in data:
        config.no_llm = bool(data["no_llm"])
    if "cache_dir" in data:
        config.cache_dir = Path(data["cache_dir"])
    if "ollama" in data and isinstance(data["ollama"], dict):
        oll = data["ollama"]
        if "base_url" in oll:
            config.ollama.base_url = oll["base_url"]
        if "api_key_env" in oll:
            config.ollama.api_key_env = oll["api_key_env"]
        if "timeout_s" in oll:
            config.ollama.timeout_s = float(oll["timeout_s"])
    if "promote" in data and isinstance(data["promote"], dict):
        prm = data["promote"]
        if "target_file" in prm:
            config.promote.target_file = Path(prm["target_file"])
        if "sections" in prm and isinstance(prm["sections"], list):
            config.promote.sections = [str(s) for s in prm["sections"]]
    if "corpora" in data and isinstance(data["corpora"], dict):
        config.corpora = _load_corpora(data["corpora"])


def _load_corpora(data: dict[str, Any]) -> dict[str, CorpusConfig]:
    """Parse the `corpora:` YAML block into a name -> CorpusConfig map."""
    out: dict[str, CorpusConfig] = {}
    for name, raw in data.items():
        if not isinstance(raw, dict):
            continue
        cfg = CorpusConfig()
        if "input_path" in raw:
            cfg.input_path = Path(raw["input_path"])
        if "output_dir" in raw:
            cfg.output_dir = Path(raw["output_dir"])
        if "include" in raw:
            cfg.include = list(raw["include"])
        if "exclude" in raw:
            cfg.exclude = list(raw["exclude"])
        if "since" in raw:
            cfg.since = date.fromisoformat(str(raw["since"]))
        if "until" in raw:
            cfg.until = date.fromisoformat(str(raw["until"]))
        if "require_dates" in raw:
            cfg.require_dates = bool(raw["require_dates"])
        if "tag_weight" in raw:
            cfg.tag_weight = float(raw["tag_weight"])
        if "top_n_tags" in raw:
            cfg.top_n_tags = int(raw["top_n_tags"])
        if "tag_weights" in raw and isinstance(raw["tag_weights"], dict):
            cfg.tag_weights = {str(k): float(v) for k, v in raw["tag_weights"].items()}
        if "min_cluster_size" in raw and raw["min_cluster_size"] is not None:
            cfg.min_cluster_size = int(raw["min_cluster_size"])
        if "umap_n_neighbors" in raw and raw["umap_n_neighbors"] is not None:
            cfg.umap_n_neighbors = int(raw["umap_n_neighbors"])
        if "promote" in raw and isinstance(raw["promote"], dict):
            prm = raw["promote"]
            cfg.promote = PromoteConfig()
            if "target_file" in prm:
                cfg.promote.target_file = Path(prm["target_file"])
            if "sections" in prm and isinstance(prm["sections"], list):
                cfg.promote.sections = [str(s) for s in prm["sections"]]
        out[str(name)] = cfg
    return out


def _apply_corpus_overrides(base: Config, name: str, corpus: CorpusConfig) -> Config:
    """Return a new Config for one corpus, inheriting unset fields from `base`."""
    cfg = Config()
    # Start by copying every base field we support.
    cfg.input_path = corpus.input_path if corpus.input_path is not None else base.input_path
    cfg.outputs = list(base.outputs)
    cfg.output_dir = (
        corpus.output_dir
        if corpus.output_dir is not None
        else base.output_dir / name
    )
    cfg.provider = base.provider
    cfg.model = base.model
    cfg.embedding_model = base.embedding_model
    cfg.ollama = base.ollama
    cfg.openai_compat = base.openai_compat
    cfg.include = corpus.include if corpus.include is not None else list(base.include)
    cfg.exclude = corpus.exclude if corpus.exclude is not None else list(base.exclude)
    cfg.since = corpus.since if corpus.since is not None else base.since
    cfg.until = corpus.until if corpus.until is not None else base.until
    cfg.top_n_themes = base.top_n_themes
    cfg.top_n_quotes = base.top_n_quotes
    cfg.spike_window_weeks = base.spike_window_weeks
    cfg.stale_window_weeks = base.stale_window_weeks
    cfg.merge_contained_phrases = base.merge_contained_phrases
    cfg.min_cluster_size = corpus.min_cluster_size if corpus.min_cluster_size is not None else base.min_cluster_size
    cfg.umap_n_neighbors = corpus.umap_n_neighbors if corpus.umap_n_neighbors is not None else base.umap_n_neighbors
    cfg.tag_weight = corpus.tag_weight if corpus.tag_weight is not None else base.tag_weight
    cfg.top_n_tags = corpus.top_n_tags if corpus.top_n_tags is not None else base.top_n_tags
    cfg.tag_weights = corpus.tag_weights if corpus.tag_weights is not None else dict(base.tag_weights)
    cfg.require_dates = corpus.require_dates if corpus.require_dates is not None else base.require_dates
    cfg.no_llm = base.no_llm
    cfg.dry_run = base.dry_run
    cfg.cache_dir = base.cache_dir
    cfg.no_cache = base.no_cache
    cfg.promote = (
        corpus.promote
        if corpus.promote is not None
        else base.promote
    )
    cfg.verbose = base.verbose
    cfg.quiet = base.quiet
    # Corpora don't nest recursively.
    return cfg


def apply_env_overrides(config: Config) -> None:
    if env := os.environ.get("TEXTHEME_OLLAMA_BASE_URL"):
        config.ollama.base_url = env
    if env := os.environ.get("TEXTHEME_OLLAMA_MODEL"):
        config.model = env
    if env := os.environ.get("TEXTHEME_OLLAMA_TIMEOUT"):
        try:
            config.ollama.timeout_s = float(env)
        except ValueError:
            pass  # Ignore malformed timeout env vars — the YAML / default wins.
    if env := os.environ.get("TEXTHEME_CACHE_DIR"):
        config.cache_dir = Path(env)
    if env := os.environ.get("TEXTHEME_TAG_WEIGHT"):
        try:
            config.tag_weight = float(env)
        except ValueError:
            pass  # Ignore malformed env vars — the YAML / default wins.
    if env := os.environ.get("TEXTHEME_LOG_LEVEL"):
        # Read by CLI when configuring logging; stored on the env, not the config.
        os.environ["TEXTHEME_LOG_LEVEL"] = env
