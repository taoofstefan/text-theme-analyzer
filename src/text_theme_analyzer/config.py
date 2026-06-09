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
from enum import Enum
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


class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    CLI = "cli"


class Provider(str, Enum):
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

    # Behavior
    no_llm: bool = False
    dry_run: bool = False
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    no_cache: bool = False

    # UX
    verbose: bool = False
    quiet: bool = False


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


def apply_env_overrides(config: Config) -> None:
    if env := os.environ.get("TEXTHEME_OLLAMA_BASE_URL"):
        config.ollama.base_url = env
    if env := os.environ.get("TEXTHEME_OLLAMA_MODEL"):
        config.model = env
    if env := os.environ.get("TEXTHEME_CACHE_DIR"):
        config.cache_dir = Path(env)
    if env := os.environ.get("TEXTHEME_LOG_LEVEL"):
        # Read by CLI when configuring logging; stored on the env, not the config.
        os.environ["TEXTHEME_LOG_LEVEL"] = env
